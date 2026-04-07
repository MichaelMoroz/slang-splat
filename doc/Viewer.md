# Viewer

## Overview

The realtime viewer is a single `spy.AppWindow` that renders the control UI into the swapchain and presents the Gaussian scene inside a docked central viewport window.

The overlay uses a right-side control panel with a menu bar:

- `File`: scene load, scene export, COLMAP import, reload, and gaussian reinitialization actions
- `View`: interface scale presets from `75%` to `200%`, with a reset action
- `Help`: `Documentation` and `About` windows
- Auxiliary windows such as `Documentation`, `About`, `Histograms`, `Training Views`, and `COLMAP Import` open as tabs in that right-side panel by default.
- The menu bar spans the full viewport width and the left control panel starts below it to avoid overlap.

`src/viewer` is split into:
- `app.py`: window lifecycle, camera input, and UI event routing
- `ui.py`: `imgui_bundle` overlay state, widget layout, and draw-data submission
- `presenter.py`: per-frame scene rendering, debug-view composition, and status text updates
- `session.py`: scene loading, renderer recreation, and training control actions
- `state.py`: persistent viewer runtime state

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

After the dataset root is selected, the viewer resolves the COLMAP reconstruction from `sparse/0`. If a COLMAP database is present it samples image names from the `images` table; otherwise it falls back to the image names stored in `images.bin`. It then walks the selected root and its subfolders until it finds the first directory that contains one of those image entries. The image folder can still be overridden manually before pressing `Import`.

Automatic RGB folder discovery skips directories whose name or full path contains `depth` so a colocated depth tree is not mistaken for the main image set.

Import-time downscale is applied before training textures are uploaded and before frame intrinsics are finalized. `Max Size` preserves aspect ratio by clamping the longer image side and scaling the shorter side to match. `Scale Factor` multiplies both dimensions uniformly. Both modes clamp to the source resolution, so the importer never upscales images.

Training-frame metadata discovery opens source images in a fixed 16-thread loader pool, which keeps large datasets from stalling on serial image-size probes while preserving the original sorted frame order.

Native training-target import uses the same 16-thread CPU loader for RGBA decode and resize work. Texture creation and `copy_from_numpy(...)` stay on the main thread, but the import path pipelines those uploads against ongoing background decode/resize work instead of waiting for the entire dataset to be processed serially.

Pointcloud initialization builds gaussians directly from the COLMAP sparse points and scales them from the median nearest-neighbor spacing multiplied by the selected coefficient. Diffused pointcloud initialization resamples sparse points with replacement and offsets each sample by `nrand3() * diffusion_radius * original_nn_distance`, where `original_nn_distance` comes from the source COLMAP point cloud, then applies the same nearest-neighbor scale initialization to the synthesized positions. Custom PLY initialization keeps the COLMAP cameras and training frames, but seeds the scene from the chosen `.ply` file instead.

`From Depth` adds a CPU-only calibration stage before the initial point cloud is built:

- RGB images are matched to depth files by relative path stem under `Depth Folder`, extension-agnostically.
- Depth maps are expected to be scalar `16-bit PNG` images.
- For each matched frame, the importer gathers up to `128` positive COLMAP `points2d_point3d_ids`, converts the COLMAP keypoints to camera-normalized coordinates, and samples the raw depth map at the matching resized-frame location.
- Correspondences are aggregated per unique COLMAP `camera_id`, not per pose, so all images sharing the same camera contribute to one calibration.
- The remap is deliberately reduced to a robust affine model:
  - `target ~= a + b * raw_depth`
- `Depth Interpretation` selects the fitted target and reverse-projection mode:
  - `Depth Is Distance`: `target` is Euclidean camera-to-point distance and reconstruction uses the camera ray.
  - `Depth Is Z-Depth`: `target` is camera-space `z` depth and reconstruction uses `screen_to_world(...)`.
- Fitting uses a ridge-regularized 2-parameter least-squares solve followed by one MAD-based outlier rejection/refit pass.
- Frames with missing or unusable depth are still imported as training views; they are only skipped when generating the depth-derived initialization cloud.
- After calibration, the importer samples a unique dataset-wide point budget across usable frames approximately proportional to each frame's valid calibrated pixel count, reverse-projects those pixels through the COLMAP camera model, and colors them from the aligned RGB image.
- The resulting positions/colors are then passed through the same nearest-neighbor scale initialization used by the point-based import modes.
- Depth maps and calibration intermediates stay on CPU and are discarded once the calibrated point cloud has been built; they are not uploaded into the runtime training textures.

## Debug Views

The loss-debug controls expose a runtime `Abs Diff Scale` slider when `View = Abs Diff`.

- The shader computes `abs(rendered - target) * scale` in linear RGB.
- `scale = 1.0` shows the raw absolute color difference.
- Higher values amplify subtle differences without changing the rendered or target views.
- Density debug views share the same range controls. `Splat Density` accumulates a soft per-pixel splat count using `sqrt(transmittance) * alpha / opacity`, while `Spatial Density` and `Screen Density` continue to normalize by 3D volume and projected ellipse area respectively.
- `Contribution Amount` visualizes the per-splat `g_SplatContribution` atomic buffer accumulated during training forward, normalized to percent of observed dataset pixels as `count / 256 / observed_pixels * 100`.
- The heatmap is logarithmic and uses dedicated `Contribution Min` and `Contribution Max` controls in percent units instead of sharing density ranges.
- `Depth Local Mismatch` tracks a front-to-back online depth estimate per pixel and visualizes the contribution-weighted local absolute depth deviation from that estimate, normalized by mean depth like `Depth Std`, with separate smooth and reject sigma-multiple controls to avoid coupling distant transparent layers.
- The renderer debug colorbar is drawn as a horizontal legend near the bottom of the docked viewport window.

## Histogram Window

`Debug -> Histograms` opens a dedicated histogram window for cached ellipse gradient components.

- The window inspects the active cached ellipse gradient accumulation mode only.
- Histogram values are computed over `log10(abs(component_gradient))`.
- Components are grouped as `roLocal`, `scale`, `quat`, `color`, and `opacity`.
- The first open requests one histogram refresh automatically.
- After that, histogram data is recomputed only when `Refresh` is pressed or another histogram action explicitly requests it.

## Cached Gradient Atomics

`Render Params` includes a `Cached Grad Atomics` selector:

- `Float Atomics`
- `Fixed Point`

`Fixed Point` is the default and uses the current cached-gradient quantization path. `Float Atomics` remains available as a fallback/reference path when the backend supports float atomics. Changing this setting only swaps the raster-backward shader/buffer path; it does not reset scene state, optimizer state, or training progress.

## Training Resolution

The `Train Setup` section exposes train downscale as a mode selector:

It also exposes a `Use Spherical Harmonics` toggle plus an `SH Start Step` slider. When enabled, projection and training use SH0+SH1 view-dependent color after that configured start step. When disabled, rendering and optimization fall back to SH0-only base color while leaving the raster hot loop unchanged.

Training background is configured separately from the viewer clear color:

- `Background Mode`: `Custom` or `Random`
- `Custom` exposes a fixed training RGB color picker and defaults to white
- `Random` re-samples the training background color per step
- `Start Densification After` now defaults to `500`.
- Refinement contribution culling is expressed in percent of observed dataset pixels, with a base default threshold of `1e-05%`.
- `Refinement Cull Decay` multiplies that threshold after each completed refinement pass and defaults to `0.995` (`0.5%` drop per pass).
- Schedule step sliders in `Train Setup`, `Learning Rates`, and `Regularization` all clamp to the current `Schedule Steps` value so breakpoint timing can be edited directly in the viewer without touching code.

- `Auto`
- manual `1x` through `16x`

Auto mode has its own `Auto Start Downscale` parameter and schedule controls:

- `Downscale Base Iters`
- `Downscale Iter Step`
- `Downscale Max Iters`

Behavior:

- Manual modes force a fixed training downscale immediately.
- Auto mode starts from `Auto Start Downscale`, then descends toward `1x`.
- Each lower factor lasts `base_iters + level_index * iter_step`.
- Training render resolution is always `ceil(native_width / N) x ceil(native_height / N)` for the effective factor.
- Loss targets are generated from the native dataset image with an exact `NxN` box filter on the GPU.
- Changing mode or crossing an auto schedule boundary recreates only the train-resolution renderer and target resources; scene state, ADAM moments, shuffle order, step counter, and pause/run state are preserved.

The panel shows both the resolved active train resolution and the current downscale status so the training renderer, loss target, and debug target view are easy to verify.

## Training Schedule

The `Optimizer` panel exposes the active training schedule directly:

- `Schedule Steps` defines the shared max-iteration budget for the LR, depth-ratio, SH warmup, and noise schedules.
- `LR Stage 1 Step` and `LR Stage 2 Step` move the two intermediate LR breakpoints.
- `Noise End Step` moves the point where random-step position noise reaches zero.
- `Depth Reg Stage 1/2/3` move the three intermediate depth-ratio regularizer breakpoints.

These breakpoint controls are regular integer sliders with a live `0..Schedule Steps` range rather than a compound multi-value slider.

## Training Metrics

The training panel shows both short-horizon and run-level throughput data.

- `Iter/s` in the status and plot sections is computed from the recent history window used for viewer plots.
- `Avg it/s` in the training table is computed from total optimizer steps divided by total accumulated active training time for the current initialized scene.
- `Time` is pause/resume-aware and accumulates only while training is active.

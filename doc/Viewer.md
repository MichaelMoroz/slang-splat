# Viewer

## Overview

The realtime viewer is a single `spy.AppWindow` that renders both the Gaussian scene and the control UI into the same swapchain image.

The overlay uses a left-side control panel with a menu bar:

- `File`: scene load, scene export, COLMAP import, reload, and gaussian reinitialization actions
- `View`: interface scale presets from `75%` to `200%`, with a reset action
- `Help`: `Documentation` and `About` windows
- The menu bar spans the full viewport width and the left control panel starts below it to avoid overlap.

`src/viewer` is split into:
- `app.py`: window lifecycle, camera input, and UI event routing
- `ui.py`: `imgui_bundle` overlay state, widget layout, and draw-data submission
- `presenter.py`: per-frame scene rendering, debug-view composition, and status text updates
- `session.py`: scene loading, renderer recreation, and training control actions
- `state.py`: persistent viewer runtime state

## Frame Flow

Each frame follows this order:

1. `SplatViewer.render(...)` calls `presenter.render_frame(...)` to render the scene or debug view into the current surface texture.
2. The overlay begins an `imgui_bundle` frame with the current surface size and frame delta.
3. The existing control panels and plots emit Dear ImGui draw lists.
4. Slangpy marshals any external font or image textures referenced by the draw data.
5. Slangpy renders the draw data directly into the same surface texture with the current command encoder.

This keeps scene rendering and UI composition in one GPU submission and avoids a second window or graphics context.

## Input Routing

Keyboard and mouse events are forwarded to the overlay first.

- If ImGui reports keyboard capture, the viewer skips its own key handling.
- If ImGui reports mouse capture, the viewer skips camera drag and scroll handling and only refreshes its cursor reference state.
- If ImGui does not capture the event, the existing camera controls run unchanged.

This avoids camera movement while using sliders, combo boxes, text inputs, plot interactions, or scrollable UI regions.

## COLMAP Import

`File -> Load COLMAP...` opens a dedicated import window instead of using an image-subdirectory selector in the main panel.

The import window collects:

- the top-level dataset root
- the image folder used for training frames
- image downscale settings: original resolution, fixed target width, or uniform scale factor
- the initialization mode: `COLMAP Pointcloud`, `Diffused Pointcloud`, or `Custom PLY`
- the nearest-neighbor radius scale coefficient used by COLMAP-based initialization
- for `Diffused Pointcloud`: synthesized point count and a dimensionless diffusion-radius multiplier

After the dataset root is selected, the viewer resolves the COLMAP reconstruction from `sparse/0`. If a COLMAP database is present it samples image names from the `images` table; otherwise it falls back to the image names stored in `images.bin`. It then walks the selected root and its subfolders until it finds the first directory that contains one of those image entries. The image folder can still be overridden manually before pressing `Import`.

Import-time downscale is applied before training textures are uploaded and before frame intrinsics are finalized. `Target Width` preserves aspect ratio by solving only for the x dimension and deriving y from the source aspect. `Scale Factor` multiplies both dimensions uniformly. Both modes clamp to the source resolution, so the importer never upscales images.

Training-frame metadata discovery opens source images in a fixed 8-thread loader pool, which keeps large datasets from stalling on serial image-size probes while preserving the original sorted frame order.

Native training-target import uses the same 8-thread CPU loader for RGBA decode and resize work. Texture creation and `copy_from_numpy(...)` stay on the main thread, but the import path pipelines those uploads against ongoing background decode/resize work instead of waiting for the entire dataset to be processed serially.

Pointcloud initialization builds gaussians directly from the COLMAP sparse points and scales them from the median nearest-neighbor spacing multiplied by the selected coefficient. Diffused pointcloud initialization resamples sparse points with replacement and offsets each sample by `nrand3() * diffusion_radius * original_nn_distance`, where `original_nn_distance` comes from the source COLMAP point cloud, then applies the same nearest-neighbor scale initialization to the synthesized positions. Custom PLY initialization keeps the COLMAP cameras and training frames, but seeds the scene from the chosen `.ply` file instead.

## Debug Views

The loss-debug controls expose a runtime `Abs Diff Scale` slider when `View = Abs Diff`.

- The shader computes `abs(rendered - target) * scale` in linear RGB.
- `scale = 1.0` shows the raw absolute color difference.
- Higher values amplify subtle differences without changing the rendered or target views.
- Density debug views share the same range controls. `Splat Density` accumulates a soft per-pixel splat count using `sqrt(transmittance) * alpha / opacity`, while `Spatial Density` and `Screen Density` continue to normalize by 3D volume and projected ellipse area respectively.
- `Contribution Amount` visualizes the per-splat `g_SplatContribution` atomic buffer accumulated during training forward. The heatmap is logarithmic and uses dedicated `Contribution Min` and `Contribution Max` controls instead of sharing density ranges.
- The renderer debug colorbar is drawn as a horizontal legend near the bottom of the main view so it stays out of the left toolkit column.

## Histogram Window

`Debug -> Histograms` opens a dedicated histogram window for cached ellipse gradient components.

- The window inspects the active cached ellipse gradient accumulation mode only.
- Histogram values are computed over `log10(abs(component_gradient))`.
- Components are grouped as `roLocal`, `scale`, `quat`, `color`, and `opacity`.
- `Auto Refresh` recomputes only when the relevant training/debug signature changes; manual refresh is available for paused inspection.

## Cached Gradient Atomics

`Render Params` includes a `Cached Grad Atomics` selector:

- `Float Atomics`
- `Fixed Point`

`Fixed Point` is the default and uses the current cached-gradient quantization path. `Float Atomics` remains available as a fallback/reference path when the backend supports float atomics. Changing this setting only swaps the raster-backward shader/buffer path; it does not reset scene state, optimizer state, or training progress.

## Training Resolution

The `Train Setup` section exposes train downscale as a mode selector:

It also exposes a `Use Spherical Harmonics` toggle. When enabled, projection and training use SH0+SH1 view-dependent color. When disabled, rendering and optimization fall back to SH0-only base color while leaving the raster hot loop unchanged.

Training background is configured separately from the viewer clear color:

- `Background Mode`: `Custom` or `Random`
- `Custom` exposes a fixed training RGB color picker and defaults to white
- `Random` re-samples the training background color per step

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

## Training Metrics

The training panel shows both short-horizon and run-level throughput data.

- `Iter/s` in the status and plot sections is computed from the recent history window used for viewer plots.
- `Avg it/s` in the training table is computed from total optimizer steps divided by total accumulated active training time for the current initialized scene.
- `Time` is pause/resume-aware and accumulates only while training is active.

# Viewer

## Overview

The realtime viewer is a single `spy.AppWindow` that renders both the Gaussian scene and the control UI into the same swapchain image.

The overlay uses a left-side control panel with a menu bar:

- `File`: scene load, COLMAP import, reload, and gaussian reinitialization actions
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
- the initialization mode: `COLMAP Pointcloud` or `Custom PLY`
- the nearest-neighbor radius scale coefficient used by COLMAP pointcloud initialization

After the dataset root is selected, the viewer recursively finds the first valid COLMAP database under that tree, then walks the root and its subfolders until it finds the first directory that contains one of the sampled COLMAP image entries. The image folder can still be overridden manually before pressing `Import`.

Pointcloud initialization builds gaussians from the COLMAP sparse points and scales them from the median nearest-neighbor spacing multiplied by the selected coefficient. Custom PLY initialization keeps the COLMAP cameras and training frames, but seeds the scene from the chosen `.ply` file instead.

## Debug Views

The loss-debug controls expose a runtime `Abs Diff Scale` slider when `View = Abs Diff`.

- The shader computes `abs(rendered - target) * scale` in linear RGB.
- `scale = 1.0` shows the raw absolute color difference.
- Higher values amplify subtle differences without changing the rendered or target views.

## Training Metrics

The training panel shows both short-horizon and run-level throughput data.

- `Iter/s` in the status and plot sections is computed from the recent history window used for viewer plots.
- `Avg it/s` in the training table is computed from total optimizer steps divided by total accumulated active training time for the current initialized scene.
- `Time` is pause/resume-aware and accumulates only while training is active.

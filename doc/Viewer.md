# Viewer

## Overview

The realtime viewer is a single `spy.AppWindow` that renders both the Gaussian scene and the control UI into the same swapchain image.

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

## Training Metrics

The training panel shows both short-horizon and run-level throughput data.

- `Iter/s` in the status and plot sections is computed from the recent history window used for viewer plots.
- `Avg it/s` in the training table is computed from total optimizer steps divided by total accumulated active training time for the current initialized scene.
- `Time` is pause/resume-aware and accumulates only while training is active.

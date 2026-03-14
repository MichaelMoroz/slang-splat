# Shader Utilities

The shader code now treats reusable GPU logic as a first-class utility submodule under `shaders/utility`.

## Layout
- `shaders/utility/math`: shared constants, numeric sanitization, quaternion helpers, matrix inverse, and camera projection types/helpers.
- `shaders/utility/splatting`: shared gaussian data structures, opacity helpers, projection geometry, and raster math helpers used by renderer stages.
- `shaders/utility/random`: reusable hash, uniform, and normal sampling helpers for shaders that need deterministic sampling utilities.
- `shaders/utility/losses`: reusable image-loss helpers for MSE, L1, and scale/opacity regularization terms.
- `shaders/utility/optimizer`: reusable optimizer-side structs and the shared ADAM update helper.
- `shaders/utility/blur`: reusable separable Gaussian blur kernels.
- `shaders/utility/prefix_sum`: reusable GPU prefix-sum kernels.
- `shaders/utility/radix_sort`: reusable GPU radix-sort kernels and indirect-dispatch helpers.

## Stage Boundaries
- `shaders/renderer/gaussian_types.slang` now owns renderer bindings, group-shared declarations, and raster-size constants.
- `shaders/renderer/gaussian_project_stage.slang` now focuses on projection/binning entry points and imports reusable splatting geometry from `shaders/utility/splatting`.
- `shaders/renderer/gaussian_raster_stage.slang` now focuses on raster forward/backward control flow and imports reusable raster math from `shaders/utility/splatting`.
- `shaders/renderer/gaussian_training_stage.slang` now keeps only the fixed-count training entry points for target downscaling, scalar loss forward reduction, image-gradient backward generation, and training/debug visualization helpers.

## Python Bindings
- `src/filter/separable_gaussian_blur.py` loads blur kernels from `shaders/utility/blur`.
- `src/scan/prefix_sum.py` loads scan kernels from `shaders/utility/prefix_sum`.
- `src/sort/radix_sort.py` loads sort kernels from `shaders/utility/radix_sort`.

`shaders/utility` is the only implementation source of truth for reusable shader logic. The old wrapper paths were removed to keep the dependency graph explicit.

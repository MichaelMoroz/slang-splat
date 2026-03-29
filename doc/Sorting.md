# Radix Sorting

`src/sort/radix_sort.py` and `shaders/utility/radix_sort/*` now use the improved `modular-refactor` radix path while preserving the current renderer integration points.

## Implementation
- Histogram and scatter passes run with `512` threads per group.
- Histograms use packed `uint` storage with two `16`-bit counters per packed entry.
- Prefix construction is wave-based and hierarchical:
- level 0 expands the packed histograms into per-digit cumulative counts,
- higher levels scan block endpoints over `256`-thread prefix blocks,
- add-back uses the shared prefix-add kernel logic already used by the prefix-sum utility.
- Indirect argument buffers encode histogram, prefix-level, scatter, and build-range dispatches plus the derived runtime parameters used by indirect sorting.
- `GPURadixSort.BUILD_RANGE_ARGS_OFFSET` remains the public offset consumed by the renderer for indirect tile-range and scanline composition work.

## Public API
- `GPURadixSort.sort_key_values(...)` sorts key/value buffers directly.
- `GPURadixSort.sort_key_values_from_count_buffer(...)` sorts up to `max_count` entries using a GPU count buffer.
- `GPURadixSort.compute_indirect_args_from_buffer_dispatch(...)` still exposes the indirect args generation path used by the renderer.

## Tests
- `tests/test_radix_sort.py` covers:
- stable direct GPU sorting against a CPU reference,
- duplicate-key stability,
- count-buffer sorting matching the direct path,
- medium and large multi-group cases,
- opt-in 32M regression coverage via `RUN_SLOW_GPU_UTILITY_TESTS=1`.

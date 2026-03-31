# GPU Prefix Sum

`src/scan/prefix_sum.py` now provides a reusable uint32 GPU prefix-sum helper.

## Purpose
- Prefix sum stays a standalone utility for GPU compaction, indirect-count handling, and accumulation tasks.
- The implementation is aligned with the improved `modular-refactor` branch utility path instead of the older recursive scan.

## Implementation
- Shader kernels live in `shaders/utility/prefix_sum/prefix_sum.slang`.
- Each block uses `256` threads and scans `512` uints by processing two values per thread with wave-prefix operations.
- Direct scans use an explicit multi-level layout over cached scratch buffers instead of recursive Python calls.
- Count-buffer scans use precomputed indirect arguments so the same kernels can scan an unknown runtime count up to a caller-supplied maximum.
- `GPUPrefixSum.prefix_scratch_elements(count)` reports the scratch element count required for the internal block-sum and block-offset hierarchies.
- `GPUPrefixSum.scan_uint(...)` supports inclusive and exclusive scans and can optionally write the final total into a one-element output buffer.
- `GPUPrefixSum.scan_uint_from_count_buffer(...)` mirrors the direct path but derives the element count from a GPU count buffer.

## Tests
- `tests/test_prefix_sum.py` covers:
- direct inclusive and exclusive scans,
- zero, single-block, and multi-block counts,
- count-buffer scans matching the direct path,
- total-out matching the CPU reduction,
- opt-in 32M regression coverage via `RUN_SLOW_GPU_UTILITY_TESTS=1`.

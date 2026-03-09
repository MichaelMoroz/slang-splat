# GPU Prefix Sum

`src/scan/prefix_sum.py` provides a reusable GPU inclusive-scan helper for `float` and `uint` buffers.

## Purpose
- Prefix sum remains a standalone utility for generic GPU compaction and accumulation tasks.
- The current fixed-count training path does not depend on it, which keeps training dispatch simpler and lets the scan code evolve independently.

## Implementation
- Shader kernels live in `shaders/utility/prefix_sum/prefix_sum.slang`.
- The implementation uses a hierarchical block scan with `256` threads per block.
- Each level scans block-local prefixes, writes one block-sum buffer, recursively scans that block-sum buffer, then adds the scanned block offsets back into the lower level.
- `GPUPrefixSum.scan_float(...)` can also write the final total sum into a separate one-element buffer for callers that need both prefixes and the overall reduction.

## Tests
- `tests/test_prefix_sum.py` covers:
  - `u32` clear,
  - single-block float scans,
  - single-block `u32` scans,
  - multi-block recursive float scans,
  - multi-block recursive `u32` scans.

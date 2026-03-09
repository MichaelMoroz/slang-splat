# GPU Prefix Sum

`src/scan/prefix_sum.py` provides a reusable GPU inclusive-scan helper for `float` and `uint` buffers.

## Purpose
- The trainer uses it to build the opacity CDF and compact dead-splat indices for the MCMC densification path.
- Keeping it outside the trainer avoids duplicating scan code inside unrelated shaders and gives the scan path its own test surface.

## Implementation
- Shader kernels live in `shaders/prefix_sum/prefix_sum.slang`.
- The implementation uses a hierarchical block scan with `256` threads per block.
- Each level scans block-local prefixes, writes one block-sum buffer, recursively scans that block-sum buffer, then adds the scanned block offsets back into the lower level.
- `GPUPrefixSum.scan_float(...)` can also write the final total sum into a separate one-element buffer, which the MCMC sampler uses as the roulette-wheel normalization constant.

## Tests
- `tests/test_prefix_sum.py` covers:
  - `u32` clear,
  - single-block float scans,
  - single-block `u32` scans,
  - multi-block recursive float scans,
  - multi-block recursive `u32` scans.

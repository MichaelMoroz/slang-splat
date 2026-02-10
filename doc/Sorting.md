# Radix Sorting

GPU radix sort implementation is copied/adapted from prior project code:
- `src/radix_sort.py`
- `shaders/radix_sort/*`

Adaptations in this repository:
- Local import paths and shader root wiring.
- API cleanup with typed Python signatures.
- Reused as a subsystem for sorting tile/depth packed splat keys.

Sorting passes:
- Histogram
- Hierarchical prefix sum
- Scatter

The renderer calls radix sort between binning and tile range construction.

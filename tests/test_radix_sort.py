from __future__ import annotations

import numpy as np

from src.sort import sort_numpy


def test_gpu_radix_sort_matches_numpy(device):
    rng = np.random.default_rng(42)
    n = 20000
    keys = rng.integers(0, np.iinfo(np.uint32).max, size=n, dtype=np.uint32)
    values = np.arange(n, dtype=np.uint32)
    sorted_keys, sorted_values = sort_numpy(device, keys, values, max_bits=32)
    expected_keys = np.sort(keys, kind="stable")
    np.testing.assert_array_equal(sorted_keys, expected_keys)
    np.testing.assert_array_equal(np.sort(sorted_values), values)

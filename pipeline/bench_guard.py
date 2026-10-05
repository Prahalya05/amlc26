"""Does the RAM-guard thread in common.py slow down numpy work? python bench_guard.py guard|noguard"""
import sys
import time

import numpy as np

if sys.argv[1] == "guard":
    import common  # noqa: F401  (starts the guard thread)
rng = np.random.default_rng(0)
t = time.time()
keep = [rng.integers(0, 10**6, 30_000_000).astype(np.int32) for _ in range(6)]  # ~0.7 GB resident
a = rng.integers(0, 180_000, 26_000_000).astype(np.int32)
o = np.argsort(a, kind="stable")
b = a[o]
c = np.unique(rng.integers(0, 2**40, 12_000_000))
print(sys.argv[1], round(time.time() - t, 2), "s", flush=True)

"""Smoke test + timing of features.py on the 20k-row sample files."""
import os
import time

import numpy as np
import pandas as pd

from common import DATA
from features import build_features

rng = np.random.default_rng(0)
n = 5000
cand = pd.DataFrame({"s1_row": np.repeat(np.arange(n), 20), "pidx": rng.integers(0, 40000, n * 20),
                     "score": rng.random(n * 20).astype(np.float32), "nkeys": 1, "rank": np.tile(np.arange(20), n)})
cand = cand.drop_duplicates(["s1_row", "pidx"])
cand["m_rid"] = 2 * 10**10 + cand.pidx
t = time.time()
out = os.path.join(DATA, "sample_feats.parquet")
build_features("sample_train", cand, out, chunk=30_000)
df = pd.read_parquet(out)
print(df.shape, f"{(time.time() - t):.1f}s")
print(df.describe().T[["mean", "min", "max"]].to_string())

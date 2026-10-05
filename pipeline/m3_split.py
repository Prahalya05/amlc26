"""M3: split TRAIN Source-1 entities into folds (by entity, never by pair).

fold 0      -> VALIDATION  (10% of S1)  : used only to measure recall / F0.5 and tune thresholds
fold 1, 2   -> MODEL TRAINING (20% of S1): candidates of these S1s (and their labels) train LightGBM
fold 3..9   -> unused for now (spare data)
The S2/S3 pools are NOT split: every S1 searches the FULL train S2+S3 pools (10.3M records),
so the number of look-alike distractors per S1 is the same as on the test set.
"""
import os

import numpy as np
import pandas as pd

from common import DATA, log

gt = pd.read_parquet(os.path.join(DATA, "train_gt_s1.parquet"))
rng = np.random.default_rng(2026)
gt["fold"] = rng.integers(0, 10, len(gt)).astype(np.int8)
gt[["s1_rid", "fold", "n_match"]].to_parquet(os.path.join(DATA, "folds.parquet"), index=False)
s = gt.groupby("fold").agg(n=("s1_rid", "size"), singleton=("n_match", lambda x: (x == 0).mean()),
                           mean_match=("n_match", "mean"))
log("folds written\n" + s.round(4).to_string())

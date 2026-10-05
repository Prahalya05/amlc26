"""Profile pair feature building on N candidate pairs from data/cand_diag10k.parquet.
Usage: python prof_feats.py N old|new
Reports setup time (tables / per-record caches) and pairwise time separately, as pairs/second."""
import cProfile
import pstats
import sys
import time

import pandas as pd

from common import DATA, log

N, which = int(sys.argv[1]), sys.argv[2]
cand = pd.read_parquet(f"{DATA}/cand_diag10k.parquet").sort_values(["s1_row", "rank"]).head(N)
pr = cProfile.Profile()
t0 = time.time()
if which == "old":
    import features as F
    s1t = F.load_side("train", [1])
    poolt = F.load_side("train", [2, 3])
    tfn, tfa = F.fit_tfidf(poolt)
    setup = time.time() - t0
    t0 = time.time()
    pr.enable()
    feats = F.pair_features(s1t, poolt, cand.s1_row.values, cand.pidx.values, tfn, tfa)
    pr.disable()
else:
    import features2 as F
    t1 = time.time()
    pr.enable()
    A = F.RecordCache("train", [1], cand.s1_row.values)
    B = F.RecordCache("train", [2, 3], cand.pidx.values)
    setup = time.time() - t0
    log(f"record caches: {A.n + B.n:,} records in {time.time() - t1:.1f}s "
        f"({(A.n + B.n) / (time.time() - t1):,.0f} records/s)")
    t0 = time.time()
    feats = F.pair_features(A, B, A.idx(cand.s1_row.values), B.idx(cand.pidx.values))
    pr.disable()
dt = time.time() - t0
log(f"[{which}] setup {setup:.0f}s; pairwise: {len(cand):,} pairs in {dt:.1f}s = {len(cand) / dt:,.0f} pairs/s, "
    f"{feats.shape[1]} features")
feats.describe().T.to_csv(f"{DATA}/../reports/prof_feats_{which}_describe.csv")
pstats.Stats(pr).sort_stats("tottime").print_stats(15)
